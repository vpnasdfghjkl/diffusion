// Generated by gencpp from file dynamic_biped/handRotationEular.msg
// DO NOT EDIT!


#ifndef DYNAMIC_BIPED_MESSAGE_HANDROTATIONEULAR_H
#define DYNAMIC_BIPED_MESSAGE_HANDROTATIONEULAR_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Vector3.h>

namespace dynamic_biped
{
template <class ContainerAllocator>
struct handRotationEular_
{
  typedef handRotationEular_<ContainerAllocator> Type;

  handRotationEular_()
    : eulerAngles()  {
    }
  handRotationEular_(const ContainerAllocator& _alloc)
    : eulerAngles(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::geometry_msgs::Vector3_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::geometry_msgs::Vector3_<ContainerAllocator> >> _eulerAngles_type;
  _eulerAngles_type eulerAngles;





  typedef boost::shared_ptr< ::dynamic_biped::handRotationEular_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dynamic_biped::handRotationEular_<ContainerAllocator> const> ConstPtr;

}; // struct handRotationEular_

typedef ::dynamic_biped::handRotationEular_<std::allocator<void> > handRotationEular;

typedef boost::shared_ptr< ::dynamic_biped::handRotationEular > handRotationEularPtr;
typedef boost::shared_ptr< ::dynamic_biped::handRotationEular const> handRotationEularConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dynamic_biped::handRotationEular_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dynamic_biped::handRotationEular_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::dynamic_biped::handRotationEular_<ContainerAllocator1> & lhs, const ::dynamic_biped::handRotationEular_<ContainerAllocator2> & rhs)
{
  return lhs.eulerAngles == rhs.eulerAngles;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::dynamic_biped::handRotationEular_<ContainerAllocator1> & lhs, const ::dynamic_biped::handRotationEular_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace dynamic_biped

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dynamic_biped::handRotationEular_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dynamic_biped::handRotationEular_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dynamic_biped::handRotationEular_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6f24fd3e9eed11c525f6da164f46e8b2";
  }

  static const char* value(const ::dynamic_biped::handRotationEular_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6f24fd3e9eed11c5ULL;
  static const uint64_t static_value2 = 0x25f6da164f46e8b2ULL;
};

template<class ContainerAllocator>
struct DataType< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dynamic_biped/handRotationEular";
  }

  static const char* value(const ::dynamic_biped::handRotationEular_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/Vector3[] eulerAngles\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::dynamic_biped::handRotationEular_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.eulerAngles);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct handRotationEular_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dynamic_biped::handRotationEular_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dynamic_biped::handRotationEular_<ContainerAllocator>& v)
  {
    s << indent << "eulerAngles[]" << std::endl;
    for (size_t i = 0; i < v.eulerAngles.size(); ++i)
    {
      s << indent << "  eulerAngles[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "    ", v.eulerAngles[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // DYNAMIC_BIPED_MESSAGE_HANDROTATIONEULAR_H